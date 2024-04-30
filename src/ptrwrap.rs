use std::ops::{Deref, DerefMut};

#[repr(transparent)]
#[derive(Debug, Clone, Copy, PartialEq, PartialOrd, Eq, Ord)]
pub(super) struct UnsafeSyncSendPtr<T>(pub *mut T);

impl<T> Deref for UnsafeSyncSendPtr<T> {
    type Target = *mut T;
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl<T> DerefMut for UnsafeSyncSendPtr<T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

unsafe impl<T: Sync> Send for UnsafeSyncSendPtr<T> {}

unsafe impl<T: Sync> Sync for UnsafeSyncSendPtr<T> {}
